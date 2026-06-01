'use client'

import { useState, useRef, useEffect } from 'react'
import Image from 'next/image'
import { createClient } from '@/lib/supabase/client'
import { useNavigation, NavigationTarget } from '../../hooks'
import { useTheme } from '@/app/contexts/ThemeContext'
import styles from './navbar.module.css'

type UserMenuProps = {
  avatarUrl: string | null
  username: string
}

type MenuItem = {
  id: NavigationTarget
  label: string
}

const staticMenuItems: MenuItem[] = [
  { id: 'settings', label: 'Profile Settings' },
]

export function UserMenu({ avatarUrl, username }: UserMenuProps) {
  const [isOpen, setIsOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const { navigate } = useNavigation()
  const {toggleTheme } = useTheme()

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const menuItems: MenuItem[] = [
    ...staticMenuItems,
    // { id: 'theme', label: theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode' },
  ]

  const handleMenuClick = (id: NavigationTarget) => {
    if (id === 'theme') {
      toggleTheme()
    } else {
      navigate(id)
    }
    setIsOpen(false)
  }

  const handleSignOut = async () => {
    const supabase = createClient()
    await supabase.auth.signOut()
    try {
      window.localStorage.removeItem('stonks:selectedPortfolioId')
    } catch {
      // Continue sign-out redirect even if storage is unavailable.
    }
    window.location.href = '/sign-in'
  }

  return (
    <div className={styles.userMenu} ref={menuRef}>
      <button
        className={styles.avatarButton}
        onClick={() => setIsOpen(!isOpen)}
      >
        {avatarUrl ? (
          <Image
            src={avatarUrl}
            alt={username}
            width={36}
            height={36}
            className={styles.avatar}
          />
        ) : (
          <div className={styles.avatarPlaceholder}>
            {username[0]?.toUpperCase() || '?'}
          </div>
        )}
      </button>

      {isOpen && (
        <div className={styles.dropdown}>
          <div className={styles.dropdownHeader}>
            <span className={styles.username}>{username}</span>
          </div>
          <div className={styles.dropdownDivider} />

          {menuItems.map((item) => (
            <button
              key={item.id}
              className={styles.dropdownItem}
              onClick={() => handleMenuClick(item.id)}
            >
              {item.label}
            </button>
          ))}

          <div className={styles.dropdownDivider} />

          <button
            className={styles.signOutItem}
            onClick={handleSignOut}
          >
            Sign Out
          </button>
        </div>
      )}
    </div>
  )
}